from abc import abstractmethod, ABCMeta


class InputStream(metaclass=ABCMeta):


    @abstractmethod
    def beginAcquistion(self): pass

    @abstractmethod
    def endAcquistion(self): pass

    @abstractmethod
    def deInit(self): pass

    @abstractmethod
    def read(self): pass

    @abstractmethod
    def getFocalLength(self): pass

    @abstractmethod
    def getCamList(self): pass