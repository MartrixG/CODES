#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
#define L 100050

string in, s;
int ma = 0;
int r[L<<1], l, rr[L<<1], ll[L<<1];

void manacher()
{
    int ctr = 0, right = 0, len = 0;
    int l = s.length();
    for(int i = 0; i < l; i++)
    {
        r[i] = (right > i) ? min(r[ctr+ctr-i], right-i+1): 1;
        while(i-r[i] >= 0 && i+r[i] < l && s[i+r[i]] == s[i-r[i]])
        {
            r[i]++;
            if(s[i+r[i]-1] == '#' && r[i]-1 > rr[i+r[i]-1])
                rr[i+r[i]-1] = r[i]-1;
            if(s[i-r[i]+1] == '#' && r[i]-1 > ll[i-r[i]+1])
                ll[i-r[i]+1] = r[i]-1;
            if(i-r[i]+1 == 0)
                rr[i+r[i]] = r[i];
            if(i+r[i] == l)
                ll[i-r[i]] = r[i];
        }
        rr[i] = max(rr[i], 1);
        ll[i] = max(ll[i], 1);
        if(s[i+r[i]-1] == '#') r[i]--;
        //if(r[i] > len) len = r[i];
        if(i+r[i]-1 > right)
        {
            right = i+r[i]-1;
            ctr = i;
        }
    }
    //printf("%d\n", len);
}

int main()
{
    for(char c = getchar(); c != '\n' && c != EOF; c = getchar())
        in += c;
    l = in.length();
    for(int i = 0; i < l-1; i++)
        s += in[i], s += '#';
    s += in[l-1]; 
    manacher();
    for(int i = 1; i < l+l-2; i += 2)
    	if(ma < rr[i]+ll[i]) ma = rr[i]+ll[i];
   	printf("%d\n", ma);
    return 0;
}
