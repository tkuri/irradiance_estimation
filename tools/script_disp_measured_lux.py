import subprocess
img_num_list = [4936, 4973, 4977, 5014, 5016, 5017, 5018, 5020, 5021, 5022, 5023, 5030, 5031]


for num in img_num_list:
    command = 'python ./tools/disp_measured_lux.py ./mydataset_result/IMG_{0}_lux.csv ./mydataset_result/IMG_{0}_input.png ./mydataset_result/IMG_{0}_lux_on_input.png'.format(num)
    subprocess.run(command)