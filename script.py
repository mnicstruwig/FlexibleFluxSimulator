print('I WAS EXECUTED')

print('Trying to persist something in the volume...')
with open('/output/myfile.txt', 'w') as f:
    f.write('The docker container wrote this!')
