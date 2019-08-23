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
	scanf("%d", &n);
	if (pd(n))
	{
		printf("1\n");
		printf("%d\n",n);
		return 0;
	}
	else
	{
		for (int i = 2; i <= n; i++)
		{
			if (pd(i) && pd(n - 2 * i))
			{
				printf("3\n%d %d %d\n", n - 2 * i, i, i);
				break; 
			}
		}
	} 
	return 0;
}