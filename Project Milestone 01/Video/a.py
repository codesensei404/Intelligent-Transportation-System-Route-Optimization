t = int(input())
ans = []
for _ in range(t):
    x = int(input())

    best_d = 1
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            best_d = max(i, x // i)

    ans.append(x - best_d)

print(*ans, sep='\n')