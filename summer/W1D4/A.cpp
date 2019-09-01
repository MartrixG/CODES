#include<cstdio>
int pd(int x)
{
	for (int i = 2; i*i <= x; i++)
	{
		if (x%i == 0) return 0;
	}
	return 1;
}
int main()
{
	int n;
	while (~scanf("%d", &n))
	{
		for (int i = n / 2; i >= 1; i--)
		{
			if (pd(i) && pd(n - i))
			{
				printf("%d %d\n", i, n - i);
				break;
			}
		}
	}
	return 0;
}