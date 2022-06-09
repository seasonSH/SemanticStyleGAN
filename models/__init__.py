# Copyright (C) 2022 ByteDance Inc.
# All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# The software is made available under Creative Commons BY-NC-SA 4.0 license
# by ByteDance Inc. You can use, redistribute, and adapt it
# for non-commercial purposes, as long as you (a) give appropriate credit
# by citing our paper, (b) indicate any changes that you've made,
# and (c) distribute any derivative works under the same license.

# THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import pprint
from .semantic_stylegan import SemanticGenerator, DualBranchDiscriminator

def make_model(args, verbose=True):
    if verbose:
        print(f"Initializing model with arguments:")
        pprint.pprint(vars(args))
    model = SemanticGenerator(args.size, args.latent, args.n_mlp, 
        channel_multiplier=args.channel_multiplier, seg_dim=args.seg_dim, 
        local_layers=args.local_layers, local_channel=args.local_channel,
        base_layers=args.base_layers, depth_layers=args.depth_layers,
        coarse_size=args.coarse_size, coarse_channel=args.coarse_channel, min_feat_size=args.min_feat_size, 
        residual_refine=args.residual_refine, detach_texture=args.detach_texture,
        transparent_dims=args.transparent_dims)
    return model