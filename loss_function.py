import tensorflow as tf
import keras.backend as K
def custom_loss(y_true, y_pred):
    # Extract "next day's price" and "today's price"
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]
    
    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]
    
    # Calculate the up/down movement (direction) for both y_true and y_pred
    y_true_diff = tf.subtract(y_true_next, y_true_tdy)
    y_pred_diff = tf.subtract(y_pred_next, y_pred_tdy)
    
    # Compare movements: if the difference is positive, it's an "UP" (1), else "DOWN" (0)
    y_true_move = tf.cast(tf.greater_equal(y_true_diff, 0), tf.float32)
    y_pred_move = tf.cast(tf.greater_equal(y_pred_diff, 0), tf.float32)
    
    # Find positions where predicted direction does not match the true direction
    direction_mismatch = tf.not_equal(y_true_move, y_pred_move)
    
    # Apply a penalty for mismatches, use tf.where to conditionally apply penalty
    penalty_factor = tf.reduce_mean(tf.abs(y_true_diff)) 
    penalty = tf.where(direction_mismatch, 
                       (tf.abs(y_true_diff - y_pred_diff) * penalty_factor), 
                       tf.zeros_like(y_true_diff))   
    # Mean squared error loss
    # mse_loss = K.mean(K.square(y_true - y_pred), axis=-1)
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    
    # Combine the MSE loss with the directional penalty
    custom_loss = mse_loss + penalty
    
    return custom_loss



