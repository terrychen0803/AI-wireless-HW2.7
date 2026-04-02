import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def simple_soft_threshold(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)


def auto_gradients(xhat, r):
    """Return the per-column average gradient of xhat with respect to r."""
    dxdr = tf.gradients(xhat, r)[0]
    dxdr = tf.reduce_mean(dxdr, 0)
    minVal = .5 / int(r.get_shape()[0])
    dxdr = tf.maximum(dxdr, minVal)
    return dxdr


def shrink_soft_threshold(r, rvar, theta):
    """
    soft threshold function
    y=sign(x)*max(0,abs(x)-theta[0]*sqrt(rvar) )*scaling
    where scaling is theta[1] (default=1)
    """
    if len(theta.get_shape()) > 0 and theta.get_shape() != (1,):
        lam = theta[0] * tf.sqrt(rvar)
        scale = theta[1]
    else:
        lam = theta * tf.sqrt(rvar)
        scale = None
    lam = tf.maximum(lam, 0)
    arml = tf.abs(r) - lam
    xhat = tf.sign(r) * tf.maximum(arml, 0)
    dxdr = tf.reduce_mean(tf.to_float(arml > 0), 0)
    if scale is not None:
        xhat = xhat * scale
        dxdr = dxdr * scale
    return xhat, dxdr
