import numpy as np
import sklearn.preprocessing as prp
# Problem 1 - Implement Min-Max scaling for grayscale image data
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    xmax = image_data.max()
    xmin = image_data.min()
    # TODO: Implement Min-Max scaling for grayscale image data
    return a+(b-a)*(image_data-xmin)/(xmax-xmin)

### DON'T MODIFY ANYTHING BELOW ###
# Test Cases
asd = normalize_grayscale(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255]))
normalize_grayscale(np.array([0, 1, 10, 20, 30, 40, 233, 244, 254,255]))


print(asd)

###############################################################


# # Solution is available in the other "solution.py" tab
# import tensorflow as tf


# def run():
#     output = None
#     logit_data = [2.0, 1.0, 0.1]
#     logits = tf.placeholder(tf.float32)
    
#     # TODO: Calculate the softmax of the logits
#     softmax = tf.nn.softmax(logits)
    
#     with tf.Session() as sess:
#         # TODO: Feed in the logit data
#         output = sess.run(softmax, feed_dict={softmax:logit_data}  )

#     return output

# print(run())

###############################################################
# # Solution is available in the other "solution.py" tab
# import numpy as np

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     exps = np.exp(x)
#     sm = exps.sum(0)
#     return exps/(sm)
#     # return exps[:None]/sm[:None]

# logits = [1.0, 2.0, 3.0]
# print(softmax(logits))

###############################################################

# import numpy as np
# w = np.array([[-0.5, 0.2, 0.1],[0.7, -0.8, 0.2]]).transpose()
# x = np.array([0.2,0.5,0.6])
# b = np.array([0.1,0.2])

# result = np.dot(x,w)+b
# print(result)
###############################################################
# import tensorflow as tf

# # Create TensorFlow object called tensor
# hello_constant = tf.constant('Hello World!')

# with tf.Session() as sess:
#     # Run the tf.constant operation in the session
#     output = sess.run(hello_constant)
#     print(output)
