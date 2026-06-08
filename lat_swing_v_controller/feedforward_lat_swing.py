import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from scipy.signal import butter, filtfilt
import numpy as np
from geometry_msgs.msg import Twist



class FeedForward(Node):
    def __init__(self):
        super().__init__('feedforward_lat_swing')

        self.subscription = self.create_subscription(
            Imu,
            '/livox/imu',
            self.feedforward_callback,
            10
        )

        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        

        #windowing params
        self.sf=0.005
        self.ws=int(0.05//self.sf)
        self.wl=int(0.5//self.sf)

        #filter params

        lowcut = 0.05  # Hz
        highcut = 40.0  # Hz
        order = 4
        nyq = 0.5 / self.sf  # Nyquist frequency
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype='band')

        #Controller param
        self.thrx=0.2
        self.thr=0.1
        self.k=3.0
        self.forw=1.0
        
        self.buffer=[]
        self.window=[]
        

    def feedforward_callback(self, msg):
        self.buffer.append([msg.linear_acceleration.y,msg.linear_acceleration.x])
        if len(self.buffer)<self.wl:
            pass
        else:
            self.window=self.buffer.copy()
            del self.buffer[-1-self.ws:-1]

            a=self.a
            b=self.b
            w=np.array(self.window)

            aYX_filt = filtfilt(b,a,w,axis=0)
            aYX_w_rms=np.mean(aYX_filt**2,axis=0)**0.5
            aYX_w_mn=np.mean(aYX_filt,axis=0)

            
            

            if aYX_w_rms[0] < self.thr:
                if aYX_w_rms[1] > self.thrx:
                    pass
                    #self.forw=np.sign(aYX_w_mn[1])
                pass
            else:
                aY_w_rms=((aYX_w_rms[0]-self.thr)*self.k)*self.forw
                twist_msg = Twist()
                twist_msg.linear.x = float(aY_w_rms)  # or use angular.z if more appropriate
                self.publisher.publish(twist_msg)
            

            





def main(args=None):
    rclpy.init(args=args)
    node = FeedForward()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
