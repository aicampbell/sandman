import sys


def readIn():
    col1 = []
    col2 = []
    with open(sys.argv[1]) as a:
        a.next()
        for line in a:
            b = line.split()
            col1.append(int(b[0])-1)
            col2.append(int(b[1])-1)
    return col1, col2

def writeTo(col1, col2):
    with open(sys.argv[2], "w+") as b:
        for i in range(len(col1)):
            b.write(str(col1[i]) + " " + str(col2[i]) + "\n") 
    


def main():
    col1, col2 = readIn()
    writeTo(col1, col2)




if __name__ == '__main__':
    main()
