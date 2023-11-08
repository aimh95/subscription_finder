import tensorflow as tf
import numpy as np
from utils.custom_utils import get_cnt_from_character_prob

def l1_loss(y_true, y_pred):
    # l = sig(abs(yi = fyi))
    err = y_true - y_pred
    loss = tf.math.reduce_mean(tf.math.abs(err))
    return loss

def objective_loss(y_true, y_pred):
    # l = sig(abs(yi = fyi))
    err = y_true - y_pred
    l_w, _ = get_cnt_from_character_prob(y_true)
    lc_w, _ = get_cnt_from_character_prob(y_pred)
    confidence_score = tf.math.divide(tf.math.subtract(l_w, tf.math.minimum(l_w, tf.math.abs(l_w-lc_w))), l_w)
    loss = tf.math.reduce_mean(tf.math.multiply(confidence_score, err))
    return loss