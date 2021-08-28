import os
import re
from utils.generate_variations import GenerateLevels
from utils.data_classes import *

# input and output folders
level_input_folder = './input/'
level_output_folder = './output/'


class GenerateVariations:

	def read_level_file(self, level_file_name):
		# read the level file and extract the objects
		print('reading the template file', level_file_name)

		all_birds = []
		all_blocks = []
		all_pigs = []
		all_tnts = []
		meta_data = []

		# use the count of the object as an identifier for the object
		object_count = 0

		with open(level_file_name) as level_file:
			for line in level_file:
				if 'Bird type' in line:
					type = re.search('type="(.*?)"', line).group(1)
					all_birds.append(Bird(type))

				elif 'Block' in line:
					object_count += 1
					type = re.search('type="(.*?)"', line).group(1)
					material = re.search('material="(.*?)"', line).group(1)
					x = re.search('x="(.*?)"', line).group(1)
					y = re.search('y="(.*?)"', line).group(1)
					rotation = re.search('rotation="(.*?)"', line).group(1)

					all_blocks.append(Block(object_count, type, material, float(x), float(y), float(rotation)))

				elif 'Platform' in line:
					object_count += 1

					type = re.search('type="(.*?)"', line).group(1)
					x = re.search('x="(.*?)"', line).group(1)
					y = re.search('y="(.*?)"', line).group(1)
					scale_x = re.search('scaleX="(.*?)"', line).group(1) if re.search('scaleX="(.*?)"',
																					  line) is not None else 1
					scale_y = re.search('scaleY="(.*?)"', line).group(1) if re.search('scaleY="(.*?)"',
																					  line) is not None else 1
					rotation = re.search('rotation="(.*?)"', line).group(1) if re.search('rotation="(.*?)"',
																						 line) is not None else 0

					all_blocks.append(
						Block(object_count, type, "", float(x), float(y), float(rotation), float(scale_x), float(scale_y)))

				elif 'Pig' in line:
					object_count += 1

					type = re.search('type="(.*?)"', line).group(1)
					x = re.search('x="(.*?)"', line).group(1)
					y = re.search('y="(.*?)"', line).group(1)
					rotation = re.search('rotation="(.*?)"', line).group(1)

					all_pigs.append(Pig(object_count, type, float(x), float(y), float(rotation)))

				elif 'TNT' in line:
					object_count += 1

					x = re.search('x="(.*?)"', line).group(1)
					y = re.search('y="(.*?)"', line).group(1)
					rotation = re.search('rotation="(.*?)"', line).group(1)

					all_tnts.append(Tnt(object_count, float(x), float(y), float(rotation)))

				else:
					if '<GameObjects>' in line or '</GameObjects>' in line or '</Level>' in line or '<Birds>' in line or '</Birds>' in line:
						continue
					else:
						meta_data.append(line)

		return all_birds, all_blocks, all_pigs, all_tnts, meta_data

	def write_level_file(self, all_birds, all_blocks, all_pigs, all_tnts, level_base_folder, file_name, meta_data):
		rounding_digits = 4

		try:
			level_file = open(level_output_folder + level_base_folder + '/' + file_name, "w")
		# level_file = open(level_output_folder  + '/' + file_name, "w")
		except:  # folder not present, create it
			os.makedirs(level_output_folder + level_base_folder)
			level_file = open(level_output_folder + level_base_folder + '/' + file_name, "w")

		# write meta data
		for line in meta_data:
			if 'Slingshot' in line:
				# write birds before the slingshot
				level_file.write('  <Birds>\n')
				for bird in all_birds:
					level_file.write('    <Bird type="%s"/>\n' % bird.type)
				level_file.write('  </Birds>\n')
			level_file.write(line)

		level_file.write('  <GameObjects>\n')

		# write pigs
		for pig in all_pigs:
			# unchanged pigs
			level_file.write('    <Pig type="%s" material="" x="%s" y="%s" rotation="%s" />\n' % (
				pig.type, str(round(pig.x, rounding_digits)), str(round(pig.y, rounding_digits)),
				str(round(pig.rotation, rounding_digits))))

		# write TNTs
		for tnt in all_tnts:
			level_file.write(
				'    <TNT type="" x="%s" y="%s" rotation="%s" />\n' % (
					str(round(tnt.x, rounding_digits)), str(round(tnt.y, rounding_digits)), str(round(tnt.rotation, rounding_digits))))

		# write blocks
		for block in all_blocks:
			# print("block", block)
			# check if platform
			if block.type == 'Platform':
				level_file.write(
					'    <Platform type="%s" material="" x="%s" y="%s" rotation="%s" scaleX="%s" scaleY="%s" />\n' % (
						block.type, str(round(block.x, rounding_digits)), str(round(block.y, rounding_digits)),
						str(round(block.rotation, rounding_digits)),
						str(round(block.scale_x, rounding_digits)),
						str(round(block.scale_y, rounding_digits))))

			# normal blocks
			else:
				level_file.write('    <Block type="%s" material="%s" x="%s" y="%s" rotation="%s" />\n' % (
					block.type, block.material, str(round(block.x, rounding_digits)), str(round(block.y, rounding_digits)),
					str(round(block.rotation, rounding_digits))))

		# close the level file
		level_file.write('  </GameObjects>\n')
		level_file.write('</Level>\n')
		level_file.close()

	def write_all_levels(self, all_birds, meta_data, generated_levels, level_base_name):
		no_of_levels_written = 0
		for generated_level in generated_levels:
			level_name = level_base_name.split('_')[0] + '_' + str("{0:02d}".format(int(level_base_name.split('_')[1]))) + '_' + str(
				"{0:02d}".format(int(level_base_name.split('_')[2]))) + '_' + str("{0:05d}".format(no_of_levels_written + 1)) + '.xml'
			level_base_folder = level_base_name.split('_')[0] + '/' + str("{0:02d}".format(int(level_base_name.split('_')[1]))) + '/' + str(
				"{0:02d}".format(int(level_base_name.split('_')[2])))
			self.write_level_file(all_birds, generated_level[0], generated_level[1], generated_level[2], level_base_folder, level_name, meta_data)
			no_of_levels_written += 1

	def main(self):
		template_files = os.listdir(level_input_folder)
		level_generator = GenerateLevels()

		for template_file in template_files:
			# read the template file
			all_birds, all_blocks, all_pigs, all_tnts, meta_data = self.read_level_file(level_input_folder + template_file)

			# print('all_birds, all_blocks, all_pigs, all_tnts, meta_data', all_birds, all_blocks, all_pigs, all_tnts, meta_data)

			template_name = template_file.rsplit('_', 1)[0]
			# create variations of the template
			generated_levels = level_generator.generate_levels_from_template(template_name, [all_blocks, all_pigs, all_tnts])

			# write the generated levels
			self.write_all_levels(all_birds, meta_data, generated_levels, template_name)


if __name__ == "__main__":
	generate_variations = GenerateVariations()
	generate_variations.main()
