#include<cstdio>
#include<cstring>
#include<iostream>
#include<cstdio>
#include<string>
using namespace std;
int f[10010];
int main()
{
    char m[10010],p[1000010];
    int n;
    scanf("%d",&n);
    while(n--)
    {
        memset(f,0,sizeof(f));
        scanf("%s%s",m,p);
        int lm=strlen(m),lp=strlen(p);
	    f[0] = -1;
	    int i = 0;
	    int j = -1;
	    while (i < lm)
	    {
		    while (j != -1 && m[i] != m[j])
		    {
			    j = f[j];
		    }
		    f[++i] = ++j;
	    }
	    j = 0;
        int ans=0;
	    for (int i = 0; i < lp; i++)
	    {
		    while (j != -1 && m[j] != p[i])
		    {
		    	j = f[j];
		    }
		    if (j == -1)
		    {
			    j = 0;
			    continue;
		    }
		    j++;
		    if (j == lm)
		    {
			    ans++;
                j=f[j];
		    }
	    }   
        printf("%d\n",ans);
    }
	return 0;
}