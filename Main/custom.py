import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
import math

class PatchEmbed(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=1, embed_dim=256, dilation=1):
        super().__init__()
        n_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.proj = nn.Conv2d(in_chans,embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=16, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        self.conv_one_one = nn.Conv2d(n_heads*2, n_heads, kernel_size=1)

    def forward(self, x, x_freq):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        print('n_head: ',self.n_heads)
        print('dim: ',self.dim)
        print('head_dim: ',self.head_dim)
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv_freq = self.qkv(x_freq)
        print('qkv shape: ',qkv.shape)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv_freq = qkv_freq.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) 
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)
        qkv_freq = qkv_freq.permute(
                2, 0, 3, 1, 4
        ) 
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_freq, k_freq, v_freq = qkv_freq[0], qkv_freq[1], qkv_freq[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        k_t_freq = k_freq.transpose(-2, -1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        dp_freq = (
           q_freq @ k_t_freq
        ) * self.scale

        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)
        attn_freq = dp_freq.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn_freq = self.attn_drop(attn_freq)
        print('Attention shape: ',attn.shape)

        attn = self.conv_one_one(torch.cat((attn,attn_freq),1))
        attn = attn.softmax(dim=-1)
        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)
        print('After Weight avg: ',x.shape)
        return x

class Attention2Dec(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x, encoder_output):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """

        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q = qkv[0]
        k = encoder_output
        v = encoder_output
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x



class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(
                x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features)

        return x


class Block(nn.Module):
    """Transformer block.

    Parameters
    ----------
    dim : int
        Embeddinig dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x, x_freq):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x), self.norm1(x_freq))
        x = x + self.mlp(self.norm2(x))

        return x

class UpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=1),
        )

        #self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def forward(self, x):
        up = self.up(x)
        return up

class VisionTransformer(nn.Module):
    """Simplified implementation of the Vision transformer.

    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            img_size=1024,
            patch_size=16,
            in_chans=1,           
            embed_dim=256,
            depth=16,
            n_heads=16,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        # shahriar start
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size//patch_size)**2
        #shahriar end

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.n_patches, embed_dim)
        )
        #self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.freq_feature = frequencyFeature(1,n_heads)
        """
        self.up = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=1),
        )
        """
        #self.up_sample = UpBlock()
        """ shahriar
        self.head = nn.Linear(embed_dim, n_classes)
        """
        #self.sim = nn.CosineSimilarity()

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        x_freq = self.freq_feature(x)
        n_samples = x.shape[0]
        print('Number of samples: ',n_samples)
        x = self.patch_embed(x)
        x_freq = self.patch_embed(x_freq)
        print('After patch embed: ',x.shape)
        x = x + self.pos_embed
        x_freq = x_freq + self.pos_embed
        print('After pos embed: ',x.shape)
        """ shahriar
        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        """
        # added by shahriar
        # add output from the high pass filter
        for block in self.blocks:
            x = block(x, x_freq)
        print('After attn: ',x.shape)
        #x = self.norm(x)
        #print('After norm: ',x.shape)
        print('patch size: ',self.patch_size)
        print('n_patches: ', self.n_patches)
        x = x.reshape(x.shape[0],self.patch_size,self.patch_size, self.n_patches)
        print('After reassemble: ',x.shape)

        
        x = x.permute(
                0, 3, 1, 2
        ) 

        for idx in range(int(math.log2(self.img_size//x.shape[3]))):
            in_size = x.shape[1]
            out_size = in_size//4
            up = UpBlock(in_size,out_size)
            x = up(x)
            print('After upsampling: ',x.shape)

        """
        in_size = x.shape[1]
        out_size = in_size//4
        up = UpBlock(in_size,out_size)
        x = up(x)
        print('After upsampling: ',x.shape)
        in_size = x.shape[1]
        out_size = in_size//4
        up = UpBlock(in_size,out_size)
        x = up(x)
        print('After upsampling: ',x.shape)
        in_size = x.shape[1]
        out_size = in_size//4
        up = UpBlock(in_size,out_size)
        x = up(x)
        print('After upsampling: ',x.shape)
        in_size = x.shape[1]
        out_size = in_size//4
        up = UpBlock(in_size,out_size)
        x = up(x)
        print('After upsampling: ',x.shape)
        in_size = x.shape[1]
        out_size = in_size//4
        up = UpBlock(in_size,out_size)
        x = up(x)
        print('After upsampling: ',x.shape)
        in_size = x.shape[1]
        out_size = in_size//4
        up = UpBlock(in_size,out_size)
        x = up(x)
        print('After upsampling: ',x.shape)
        """

        """
        similarity_map = np.zeros((self.n_patches,self.n_pathces))

        #added by shahriar
        for idx1 in self.n_pathces:
            for idx2 in self.n_pathces:
                sim_sum = 0
                for neighbor in range(9):                   
                    sim_sum += self.sim(x[:,:,]) # complete this
                similarity_map[idx1,idx2] = sim_sum/9   
        """ 

        """ shahriar
        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)
        """
        return x   

class DecoderModule(nn.Module):
    def __init__(
            self,
            img_size=384,
            patch_size=16,
            in_chans=1,           
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        """ shahriar
        self.head = nn.Linear(embed_dim, n_classes)
        """

    def forward(self, x, encoder_output):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed(x)
        """ shahriar
        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        """
        # added by shahriar
        # add output from the high pass filter
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)


        """ shahriar
        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)
        """

        return x   
class frequencyFeature(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size=3
        ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding = 'same')
    def forward(self, im):
        
        #option 1
        f_threshold = 10    
        filtermask = torch.ones(im.shape[2],im.shape[3]).to("cuda")
        for i in range(f_threshold+1):
            for j in range(f_threshold+1):
                if j<=f_threshold - i:
                    filtermask[i,j] = 0
        x = dct.dct_2d(im)
        x = x * filtermask
        
        x = dct.idct_2d(x)

        #x = self.conv1(x)
        print('after freq conv: ',x.shape)
        return x
        """
        #option 2
        x = im - cv2.GaussianBlur(im, (21, 21), 3)+127
        return x
        """

class MaskEstimator(nn.Module):
    def __init__(
        self,
        encoder = VisionTransformer(),
        decoder = DecoderModule()
        ):
        super.__init__()
    def forward(self, im, target_im):
        H, W = im.size(2), im.size(3)
        encoder_output = self.encoder(im)
        decoder_output = self.decoder(target_im, encoder_output)
        mask = F.interpolate(decoder_output, size=(H, W), mode="bilinear")


