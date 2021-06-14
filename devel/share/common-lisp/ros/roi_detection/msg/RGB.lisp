; Auto-generated. Do not edit!


(cl:in-package roi_detection-msg)


;//! \htmlinclude RGB.msg.html

(cl:defclass <RGB> (roslisp-msg-protocol:ros-message)
  ((m_rgb
    :reader m_rgb
    :initarg :m_rgb
    :type (cl:vector cl:float)
   :initform (cl:make-array 0 :element-type 'cl:float :initial-element 0.0)))
)

(cl:defclass RGB (<RGB>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <RGB>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'RGB)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name roi_detection-msg:<RGB> is deprecated: use roi_detection-msg:RGB instead.")))

(cl:ensure-generic-function 'm_rgb-val :lambda-list '(m))
(cl:defmethod m_rgb-val ((m <RGB>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader roi_detection-msg:m_rgb-val is deprecated.  Use roi_detection-msg:m_rgb instead.")
  (m_rgb m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <RGB>) ostream)
  "Serializes a message object of type '<RGB>"
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'm_rgb))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (cl:let ((bits (roslisp-utils:encode-single-float-bits ele)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream)))
   (cl:slot-value msg 'm_rgb))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <RGB>) istream)
  "Deserializes a message object of type '<RGB>"
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'm_rgb) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'm_rgb)))
    (cl:dotimes (i __ros_arr_len)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:aref vals i) (roslisp-utils:decode-single-float-bits bits))))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<RGB>)))
  "Returns string type for a message object of type '<RGB>"
  "roi_detection/RGB")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'RGB)))
  "Returns string type for a message object of type 'RGB"
  "roi_detection/RGB")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<RGB>)))
  "Returns md5sum for a message object of type '<RGB>"
  "23c5d650889550da7f52653940412067")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'RGB)))
  "Returns md5sum for a message object of type 'RGB"
  "23c5d650889550da7f52653940412067")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<RGB>)))
  "Returns full string definition for message of type '<RGB>"
  (cl:format cl:nil "float32[] m_rgb~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'RGB)))
  "Returns full string definition for message of type 'RGB"
  (cl:format cl:nil "float32[] m_rgb~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <RGB>))
  (cl:+ 0
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'm_rgb) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ 4)))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <RGB>))
  "Converts a ROS message object to a list"
  (cl:list 'RGB
    (cl:cons ':m_rgb (m_rgb msg))
))
