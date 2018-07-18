import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import carla.image_converter as ic

class SegConv(object):
    """docstring for SegConv."""
    def __init__(self):
        super(SegConv, self).__init__()
        rospy.on_shutdown(self.shutdown_hook)
        rospy.init_node('segconv')

        self.frequency = 10
        self.rate = rospy.Rate(self.frequency)
        self.bridge = CvBridge()

        self.seg_im_pub_ = rospy.Publisher('/segmented_image', Image, queue_size=1)

        self.raw_seg_im_sub_ = rospy.Subscriber('/camera_seg/image_raw', Image, self.seg_cb)

    def shutdown_hook(self):
        """
            Callback function invoked when initiating a node shutdown. Called
            before actual shutdown occurs.
        """
        print("Shutting down")
        # sys.exit()
        # os.system('kill %d' % os.getpid())

    def return_rate(self):
        return self.rate

    def seg_cb(self, data):
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
          print(e)

        new_img = ic.labels_to_cityscapes_palette(np.asarray(cv_image))
        print(cv_image)
        cv2.imshow("test", new_img)
        cv2.waitKey(3)
        # msg = Image()
        # msg.header.stamp = rospy.Time.now()
        # msg.data = new_img
        # self.seg_im_pub_.publish(msg)

def main():
    try:
        seg_conv = SegConv()
        node_rate = seg_conv.return_rate()

        while rospy.is_shutdown() is not True:
            node_rate.sleep()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
