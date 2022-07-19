import numpy as np
import os
import csv
import shutil
dataset_name = "waterbirds_v1.0"


def copy_files(root_dir,output_folder_name,metadata):
    for row in metadata:
        folder_name = row[1].split("/")[0]
        image_name = row[1].split("/")[1]
        #print(folder_name,image_name)
        orig_file_path = os.path.join(root_dir,folder_name,image_name)
        new_file_path = os.path.join(output_folder_name,folder_name,image_name)
        if not os.path.isdir(os.path.join(output_folder_name,folder_name)):
            os.mkdir(os.path.join(output_folder_name,folder_name))
        shutil.copyfile(orig_file_path,new_file_path)

    shutil.copyfile(os.path.join(root_dir,"RELEASE_v1.0.txt"), os.path.join(output_folder_name,"RELEASE_v1.0.txt"))
    print("DONE with copy files")




def create_dataset_via_index(index_file,output_folder_name,root_dir):
    a = np.load(index_file)
    a = sorted(a.astype(int))
    header = []
    new_content = []
    al_dir = os.path.join(root_dir, output_folder_name,dataset_name)
    if not os.path.isdir(os.path.join(root_dir, output_folder_name)):
        os.mkdir(os.path.join(root_dir, output_folder_name))

    if not os.path.isdir(al_dir):
        os.mkdir(al_dir)
    csv_file_name = os.path.join(root_dir, dataset_name, "metadata.csv")
    with open(csv_file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        i = 0
        for row in csv_reader:
            if i == 0:
                header = row
            if i >= 1 and i < 400000:
                # print(row[0])
                if (int(row[0]) in a):
                    new_content.append(row)
            i += 1


    print("START WRITING CSV FILE")
    with open(os.path.join(al_dir, "metadata.csv"), "w", newline='') as f:
        write = csv.writer(f)
        write.writerow(header)
        write.writerows(new_content)

    print("DONE with copy CSV\n")

    copy_files(os.path.join(root_dir,dataset_name),al_dir,new_content)



def create_dataset_via_csv(index_file,output_folder_name,root_dir):
    temp_haeder = []
    temp_content = {}
    a = []
    with open(index_file,'r') as f:
        reader = csv.reader(f)
        j = 0
        for row in reader:
            if j == 0:
                temp_haeder = row
            else:
                a.append(int(row[1]))
                temp_content.update({str(int(row[1])):row[2]})
            j+=1
    print(temp_content)



    header = []
    new_content = []
    al_dir = os.path.join(root_dir, output_folder_name,dataset_name)
    if not os.path.isdir(os.path.join(root_dir, output_folder_name)):
        os.mkdir(os.path.join(root_dir, output_folder_name))

    if not os.path.isdir(al_dir):
        os.mkdir(al_dir)
    csv_file_name = os.path.join(root_dir, dataset_name, "metadata.csv")
    with open(csv_file_name, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        i = 0
        for row in csv_reader:
            if i == 0:
                header = row
            if i >= 1 and i < 400000:
                # print(row[0])
                if (int(row[0]) in a):
                    row.append(temp_content[str(int(row[0]))])
                    new_content.append(row)
            i += 1

    header.append(temp_haeder[2])
    print(header)

    print("START WRITING CSV FILE")
    with open(os.path.join(al_dir, "metadata.csv"), "w", newline='') as f:
        write = csv.writer(f)
        write.writerow(header)
        write.writerows(new_content)

    print("DONE with copy CSV\n")

    copy_files(os.path.join(root_dir,dataset_name),al_dir,new_content)




if __name__ == '__main__':
    #change root_dir to your wilds_data path
    root_dir = os.path.join("/hdd2", "wilds_data")
    # create_dataset_via_index("high_idx.npy","high_index",root_dir)
    # create_dataset_via_index("low_idx.npy","low_index",root_dir)
    create_dataset_via_csv("high.csv","high",root_dir)
    create_dataset_via_csv("low.csv", "low", root_dir)

