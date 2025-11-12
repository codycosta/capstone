# pseudo-example for A→B navigation
path = plan_path(start=(0,0), goal=(30,10), obstacles=obstacle_map)
for x, y in path:
    client.moveToPositionAsync(x, y, -5, 3).join()
