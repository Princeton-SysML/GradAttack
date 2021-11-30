"""
The decode step for InstaHide and MixUp is based on Carlini et al.'s attack: Is Private Learning Possible with Instance Encoding? (https://arxiv.org/pdf/2011.05315.pdf)
The implementation heavily relies on their code: https://github.com/carlini/privacy/commit/28b8a80924cf3766ab3230b5976388139ddef295 
"""

import argparse
import math
import os

import jax
import jax.experimental.optimizers
import jax.numpy as jn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='gradattack training')
    parser.add_argument('--dir', default=None, type=str)
    parser.add_argument('--dest_dir', default=None, type=str)
    parser.add_argument('--num_batches', default=50, type=int)
    parser.add_argument('--instahide', action='store_true')
    parser.add_argument('--k', default=4, type=int)
    parser.add_argument('--dim', default=32, type=int)
    parser.add_argument('--num_img', default=50, type=int)
    parser.add_argument('--num_epoch', default=20, type=int)
    args = parser.parse_args()

    return args


def toimg(x, ori=True):
    """Convert a list of images to a single patched image for better illustration
    """
    if ori is True:
        img = (x - x.min()) / (x.max() - x.min())
        img *= 255
    else:
        img = (x + 1) * 127.5
        img = np.clip(img, 0, 255)
    if len(x) % 10 != 0:
        pad_size = int(math.ceil(len(img) / 10) * 10) - len(img)
        img = np.append(img, np.zeros((pad_size, *img[0].shape)), axis=0)
    img = np.reshape(img, (len(img) // 10, 10, args.dim, args.dim, 3))
    img = np.concatenate(img, axis=2)
    img = np.concatenate(img, axis=0)
    img = Image.fromarray(np.array(img, dtype=np.uint8))
    return img


def explained_variance(I,
                       private_images,
                       lambdas,
                       encoded_images,
                       public_to_private,
                       return_mat=False):
    # private images: args.num_epochx32x32x3
    # encoded images: args.num_img*args.num_epoch x32x32x3

    public_to_private = jax.nn.softmax(public_to_private, axis=-1)

    # Now combine them together to get the variance we can explain
    merged = np.zeros(encoded_images.shape)
    for ik in range(k):
        merged += lambdas[:, ik][:, None, None, None] * jn.dot(
            public_to_private[ik], private_images.reshape(
                (args.num_img, -1))).reshape(
                    (args.num_img * args.num_epoch, args.dim, args.dim, 3))

    # And now get the variance we can't explain.
    # This is the contribution of the public images.
    # We want this value to be small.

    def keep_smallest_abs(xx1, xx2):
        t = 0
        which = (jn.abs(xx1 + t) < jn.abs(xx2 + t)) + 0.0
        return xx1 * which + xx2 * (1 - which)

    if args.instahide:

        xx1 = jn.abs(encoded) - merged
        xx2 = -(jn.abs(encoded) + merged)

        xx = keep_smallest_abs(xx1, xx2)
        unexplained_variance = xx

        if return_mat:
            return unexplained_variance, xx1, xx2

        extra = (1 - jn.abs(private_images)).mean() * .05

        return extra + (unexplained_variance**2).mean()
    else:
        diff = encoded - merged
        if return_mat:
            return diff, diff, diff

        extra = (1 - jn.abs(private_images)).mean() * .05
        return extra + (diff**2).mean()


def setup(encoded_p=None, lambdas_p=None, using_p=None):
    global private, encoded, lambdas, using

    # Load all the things we've made.
    if encoded_p is None:
        encoded = np.load("data/encryption.npy")
        labels = np.load("data/label.npy")
        using = np.load("data/predicted_pairings_80.npy", allow_pickle=True)
        lambdas = list(
            np.load("data/predicted_lambdas_80.npy", allow_pickle=True))
        for x in lambdas:
            while len(x) < 2:
                x.append(0)
        lambdas = np.array(lambdas)
    else:
        encoded = encoded_p
        lambdas = lambdas_p
        using = using_p

    # Construct the mapping
    public_to_private_new = np.zeros(
        (k, args.num_img * args.num_epoch, args.num_img))

    for i, row in enumerate(using):
        for j, b in enumerate(row):
            public_to_private_new[j][i][int(b)] = 1e9
    using = public_to_private_new


def loss(private, lams, I):
    return explained_variance(I, private, lams, jn.array(encoded),
                              jn.array(using))


def make_loss():
    global vg
    vg = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))


def run(mode='grad'):
    priv = np.zeros((args.num_img, args.dim, args.dim, 3))
    uusing = np.array(using)
    lams = np.array(lambdas)

    # Use Adam, because thinking hard is overrated we have magic pixie dust.
    init_1, opt_update_1, get_params_1 = jax.experimental.optimizers.adam(.01)

    @jax.jit
    def update_1(i, opt_state, gs):
        return opt_update_1(i, gs, opt_state)

    opt_state_1 = init_1(priv)

    # 1000 iterations of gradient descent is probably enough
    for i in range(1000):
        value, grad = vg(priv, lams, i)

        if i % 100 == 0:
            var, _, _ = explained_variance(0,
                                           priv,
                                           jn.array(lambdas),
                                           jn.array(encoded),
                                           jn.array(using),
                                           return_mat=True)
            print('unexplained min/max', var.min(), var.max())
        opt_state_1 = update_1(i, opt_state_1, grad[0])
        priv = opt_state_1.packed_state[0][0]

    priv -= np.min(priv, axis=(1, 2, 3), keepdims=True)
    priv /= np.max(priv, axis=(1, 2, 3), keepdims=True)

    priv *= 2
    priv -= 1

    # Finally save the stored values

    plt.imshow(toimg(priv))
    plt.axis('off')
    # dest_dir = os.path.join('decode', SUB_ROOT_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    if mode == 'grad':
        plt.savefig(os.path.join(dest_dir, 'grad_decode.png'),
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0)
        np.save(os.path.join(dest_dir, 'grad_decode.npy'), priv)
    else:
        plt.savefig(os.path.join(dest_dir, 'vanilla_decode.png'),
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0)
        np.save(os.path.join(dest_dir, 'vanilla_decode.npy'), priv)


if __name__ == "__main__":
    MODE = 'trained'

    args = parse_args()
    ROOT_DIR = args.dir

    SUB_ROOT_DIR = ROOT_DIR
    num_batches = args.num_batches
    k = args.k

    # Load dumped results.
    # Note: to evaluate the upper-bound for privacy leakage, we are assuming a VERY strong (and unrealisitc) attacker, who knows the encoding mapping and mixing coefficients!
    lams = []
    selects = []
    grad_encodes = []
    vanilla_encodes = []
    originals = []
    for epoch in range(args.num_epoch):
        try:
            lam = np.loadtxt(
                os.path.join(ROOT_DIR, f'Epoch_{epoch}/epoch_lams.txt'))
            select = np.loadtxt(
                os.path.join(ROOT_DIR, f'Epoch_{epoch}/epoch_selects.txt'))
            lams.append(lam)
            selects.append(select)
            for bidx in range(num_batches):
                grad_encode = np.load(
                    os.path.join(ROOT_DIR,
                                 f'Epoch_{epoch}/{bidx}/reconstructed.npy'))
                if len(grad_encode.shape) == 3:
                    grad_encode = np.reshape(grad_encode,
                                             (1, args.dim, args.dim, 3))
                vanilla_encode = np.load(
                    os.path.join(ROOT_DIR,
                                 f'Epoch_{epoch}/{bidx}/transformed.npy'))
                if len(vanilla_encode.shape) == 3:
                    vanilla_encode = np.reshape(vanilla_encode,
                                                (1, args.dim, args.dim, 3))

                grad_encodes.extend(list(grad_encode))
                vanilla_encodes.extend(list(vanilla_encode))

                if epoch == 0:
                    original = np.load(
                        os.path.join(ROOT_DIR,
                                     f'Epoch_{epoch}/{bidx}/original.npy'))
                    if len(original.shape) == 3:
                        original = np.reshape(original,
                                              (1, args.dim, args.dim, 3))
                    originals.extend(list(original))
        except:
            continue

    lams = np.concatenate(lams)
    selects = np.concatenate(selects)
    grad_encodes = np.stack(grad_encodes)
    vanilla_encodes = np.stack(vanilla_encodes)
    originals = np.stack(originals)

    # Save original private images
    plt.imshow(toimg(originals))
    plt.axis('off')
    dest_dir = os.path.join(args.dest_dir, SUB_ROOT_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    plt.savefig(os.path.join(dest_dir, 'originals.png'),
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)
    np.save(os.path.join(dest_dir, 'originals.npy'), originals)

    # Save decoded images from gradients
    setup(encoded_p=vanilla_encodes, lambdas_p=lams, using_p=selects)
    make_loss()
    run(mode='vanilla')

    # Save decoded images from gradients
    setup(encoded_p=grad_encodes, lambdas_p=lams, using_p=selects)
    make_loss()
    run(mode='grad')
