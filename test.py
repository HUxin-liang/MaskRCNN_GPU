class te():
    def __init__(self, architecture):
        super(te, self).__init__()
        assert architecture in ['resnet50', 'resnet101']
        self.layers = [3, 4, {'resnet50': 6, 'resnet101': 23}[architecture], 3]
    def print(self):
        print(self.layers)


a = te('resnet101')
print(a)
a.print()