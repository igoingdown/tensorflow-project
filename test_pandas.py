with open("test_file.txt", "w") as f:
    for i in range(10):
        for j in range(10):
            f.write(str(i))
            if j != 9:
                f.write("\t")
        f.write("\n")
