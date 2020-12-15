class IO() :
    def __init__(self, path) :
        with open(path, 'r') as openfile :
            self.texts = openfile.readlines()
        self.timests, self.samples = self.get(text)
        print("We get times {} and samples {} data".format(len(self.timests), len(self.samples)))

    def get(self, texts) :
        return [texts[i][:-1] for i in range(len(texts)) if i%3==0], [texts[i][:-1] for i in range(len(texts)) if i%3==1]