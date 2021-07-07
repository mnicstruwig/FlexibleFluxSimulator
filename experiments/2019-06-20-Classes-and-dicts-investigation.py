class MyClass(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_value(self, which="x"):
        return self.__dict__[which]
