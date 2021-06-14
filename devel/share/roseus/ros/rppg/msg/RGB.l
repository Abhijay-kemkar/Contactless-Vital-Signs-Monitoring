;; Auto-generated. Do not edit!


(when (boundp 'rppg::RGB)
  (if (not (find-package "RPPG"))
    (make-package "RPPG"))
  (shadow 'RGB (find-package "RPPG")))
(unless (find-package "RPPG::RGB")
  (make-package "RPPG::RGB"))

(in-package "ROS")
;;//! \htmlinclude RGB.msg.html


(defclass rppg::RGB
  :super ros::object
  :slots (_m_rgb ))

(defmethod rppg::RGB
  (:init
   (&key
    ((:m_rgb __m_rgb) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _m_rgb __m_rgb)
   self)
  (:m_rgb
   (&optional __m_rgb)
   (if __m_rgb (setq _m_rgb __m_rgb)) _m_rgb)
  (:serialization-length
   ()
   (+
    ;; float32[] _m_rgb
    (* 4    (length _m_rgb)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _m_rgb
     (write-long (length _m_rgb) s)
     (dotimes (i (length _m_rgb))
       (sys::poke (elt _m_rgb i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _m_rgb
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _m_rgb (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _m_rgb i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get rppg::RGB :md5sum-) "23c5d650889550da7f52653940412067")
(setf (get rppg::RGB :datatype-) "rppg/RGB")
(setf (get rppg::RGB :definition-)
      "float32[] m_rgb

")



(provide :rppg/RGB "23c5d650889550da7f52653940412067")


