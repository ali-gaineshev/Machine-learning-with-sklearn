def main():
    with open('./data/dataset2.csv', 'r') as csv_file:
        lines = csv_file.readlines()


    with open('output.xyz', 'w') as xyz_file:
        for line in lines:
            
            x, y, z = line.strip().split(',')
            xyz_file.write(f"{x} {y} {z}\n")

main()
