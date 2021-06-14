
(cl:in-package :asdf)

(defsystem "roi_detection-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "RGB" :depends-on ("_package_RGB"))
    (:file "_package_RGB" :depends-on ("_package"))
  ))