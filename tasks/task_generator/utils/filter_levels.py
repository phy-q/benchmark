# importing os module
import os
from shutil import copy

level_base_dir = '../generated_levels/third generation - 200 levels filtered/'
level_list_to_delete = 'levels_to_delete.csv'


# Function to rename multiple files
def main():
	with open(level_list_to_delete) as fp:
		lines = fp.readlines()
		for line in lines[1:]:  # skip the header
			level_file_name = line.split(',')[1]
			# make the level path from the level file name
			level_file_path = level_base_dir + level_file_name.split('_')[0] + '/' + level_file_name.split('_')[1] + '/' + level_file_name.split('_')[2] + '/' + level_file_name
			print('removing the level', level_file_path)
			os.remove(level_file_path)

	print('removed ', len(lines) - 1, 'levels')


# i = 1
# # "{0:05d}".format(i)
# for filename in os.listdir(source_dir):
#     # new_file_name = filename.split('_')[0] + '_'+ filename.split('_')[1] + '_0_7_3'
#     new_file_name = filename.split('_')[0] + '_1_0_7_3'
#     print(new_file_name)
#
#     dst = destination_dir + new_file_name + ".xml"
#     src = source_dir + '/' + filename
#
#     os.rename(src, dst)
#     i += 1


# Driver Code
if __name__ == '__main__':
	# Calling main() function
	main()
