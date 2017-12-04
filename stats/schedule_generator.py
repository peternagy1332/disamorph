start_learning_rate = float(input("Start learning rate:"))
start_until_global_step = int(input("Start until global step:"))
schedule_rows = int(input("Schedule rows:"))
i=2
for _ in range(schedule_rows):
    print('    - {learning_rate: %.8g, until_global_step: %d}' % (start_learning_rate, start_until_global_step))
    start_until_global_step = int(start_until_global_step * i)
    start_learning_rate/=2
    i-=1/schedule_rows