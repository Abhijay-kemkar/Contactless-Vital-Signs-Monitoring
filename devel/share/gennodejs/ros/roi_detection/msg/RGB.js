// Auto-generated. Do not edit!

// (in-package roi_detection.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class RGB {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.m_rgb = null;
    }
    else {
      if (initObj.hasOwnProperty('m_rgb')) {
        this.m_rgb = initObj.m_rgb
      }
      else {
        this.m_rgb = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type RGB
    // Serialize message field [m_rgb]
    bufferOffset = _arraySerializer.float32(obj.m_rgb, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type RGB
    let len;
    let data = new RGB(null);
    // Deserialize message field [m_rgb]
    data.m_rgb = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.m_rgb.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'roi_detection/RGB';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '23c5d650889550da7f52653940412067';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[] m_rgb
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new RGB(null);
    if (msg.m_rgb !== undefined) {
      resolved.m_rgb = msg.m_rgb;
    }
    else {
      resolved.m_rgb = []
    }

    return resolved;
    }
};

module.exports = RGB;
