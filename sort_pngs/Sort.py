import os


def rename_jpg_files(directory):
    # Get a list of all .jpg files in the directory
    jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]

    # Sort the files to ensure they are processed in order
    jpg_files.sort()

    # Step 1: Rename all files to temporary names
    temp_files = []
    for index, filename in enumerate(jpg_files):
        temp_name = f"temp_{index}.jpg"
        old_path = os.path.join(directory, filename)
        temp_path = os.path.join(directory, temp_name)

        os.rename(old_path, temp_path)
        temp_files.append(temp_name)
        print(f"Renamed {filename} to {temp_name}")

    # Step 2: Rename temporary files to final names
    for index, temp_name in enumerate(temp_files, start=0):
        new_name = f"img_{index}.jpg"
        temp_path = os.path.join(directory, temp_name)
        new_path = os.path.join(directory, new_name)

        os.rename(temp_path, new_path)
        print(f"Renamed {temp_name} to {new_name}")


# Example usage
directory_path = "E:\\dev\\yolo-opencv-detector-main\\images"  # Replace with your directory path
rename_jpg_files(directory_path)