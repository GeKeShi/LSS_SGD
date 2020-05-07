keys_group_gpu = [-0.00384536,-0.0026085,-0.00113193,-0.00021912,0.00022157,0.00113283,0.00261214,0.00389169]
grad = 0.00031104538
tmp = abs(keys_group_gpu[0] - grad)
tmp_index = 0
keys_group_gpu[0] = 1
for i in range(8):
    if(abs(keys_group_gpu[i] - grad) < tmp):
        tmp = abs(keys_group_gpu[i] - grad)
        keys_group_gpu[i] = 1
        keys_group_gpu[tmp_index] = 0
        tmp_index = i
    else:
        keys_group_gpu[i] = 0

print(keys_group_gpu)