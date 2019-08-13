#include<bits/stdc++.h>
#define N 100000
using namespace std;
int n,st,rt,cnt,tot,cur[N+5],Void[N+5];
const double alpha=0.75;
struct Scapegoat
{
    int Son[2],Exist,Val,Size,Fac;
}node[N+5];
inline char tc()
{
    static char ff[100000],*A=ff,*B=ff;
    return A==B&&(B=(A=ff)+fread(ff,1,100000,stdin),A==B)?EOF:*A++;
}
inline void read(int &x)
{
    x=0;int f=1;char ch;
    while(!isdigit(ch=tc())) if(ch=='-') f=-1;
    while(x=(x<<3)+(x<<1)+ch-'0',isdigit(ch=tc()));
    x*=f;
}
inline void write(int x)
{
    if(x<0) putchar('-'),x=-x;
    if(x>9) write(x/10);
    putchar(x%10+'0');
}
inline void Init()
{
    tot=0;
    for(register int i=N-1;i;--i) Void[++tot]=i;
}
inline bool balance(int x)
{
    return (double)node[x].Fac*alpha>(double)max(node[node[x].Son[0]].Fac,node[node[x].Son[1]].Fac);
}
inline void Build(int x)
{
    node[x].Son[0]=node[x].Son[1]=0,node[x].Size=node[x].Fac=1;
}
inline void Insert(int &x,int val)
{
    if(!x)
    {
        x=Void[tot--],node[x].Val=val,node[x].Exist=1,Build(x);
        return;
    }
    ++node[x].Size,++node[x].Fac;
    if(val<=node[x].Val) Insert(node[x].Son[0],val);
    else Insert(node[x].Son[1],val);
}
inline void PushUp(int x)
{
    node[x].Size=node[node[x].Son[0]].Size+node[node[x].Son[1]].Size+1,node[x].Fac=node[node[x].Son[0]].Fac+node[node[x].Son[1]].Fac+1;	
}
inline void Traversal(int x)
{
    if(!x) return;
    Traversal(node[x].Son[0]);
    if(node[x].Exist) cur[++cnt]=x;
    else Void[++tot]=x;
    Traversal(node[x].Son[1]);
}
inline void SetUp(int l,int r,int &x)
{
    int mid=l+r>>1;x=cur[mid];
    if(l==r)
    {
        Build(x);
        return;
    }
    if(l<mid) SetUp(l,mid-1,node[x].Son[0]);
    else node[x].Son[0]=0;
    SetUp(mid+1,r,node[x].Son[1]),PushUp(x);
}
inline void ReBuild(int &x)
{
    cnt=0,Traversal(x);
    if(cnt) SetUp(1,cnt,x);
    else x=0;
}
inline void check(int x,int val)
{
    int s=val<=node[x].Val?0:1;
    while(node[x].Son[s])
    {
        if(!balance(node[x].Son[s])) 
        {
            ReBuild(node[x].Son[s]);
            return;
        }
        x=node[x].Son[s],s=val<=node[x].Val?0:1;
    }
}
inline int get_rank(int v)
{
    int x=rt,rk=1;
    while(x)
    {
        if(node[x].Val>=v) x=node[x].Son[0];
        else rk+=node[node[x].Son[0]].Fac+node[x].Exist,x=node[x].Son[1];
    }
    return rk;
}
inline int get_val(int rk)
{
    int x=rt;
    while(x)
    {
        if(node[x].Exist&&node[node[x].Son[0]].Fac+1==rk) return node[x].Val;
        else if(node[node[x].Son[0]].Fac>=rk) x=node[x].Son[0];
        else rk-=node[x].Exist+node[node[x].Son[0]].Fac,x=node[x].Son[1];
    }
}
inline void Delete(int &x,int rk)
{
    if(node[x].Exist&&!((node[node[x].Son[0]].Fac+1)^rk)) 
    {
        node[x].Exist=0,--node[x].Fac;
        return;
    }
    --node[x].Fac;
    if(node[node[x].Son[0]].Fac+node[x].Exist>=rk) Delete(node[x].Son[0],rk);
    else Delete(node[x].Son[1],rk-node[x].Exist-node[node[x].Son[0]].Fac);
}
inline void del(int v)
{
    Delete(rt,get_rank(v));
    if((double)node[rt].Size*alpha>(double)node[rt].Fac) ReBuild(rt);
}
int main()
{
    for(read(n),Init();n;--n)
    {
        int op,x;read(op),read(x);
        switch(op)
        {
            case 1:st=rt,Insert(rt,x),check(st,x);break;
            case 2:del(x);break;
            case 3:write(get_rank(x)),putchar('\n');break;
            case 4:write(get_val(x)),putchar('\n');break;
            case 5:write(get_val(get_rank(x)-1)),putchar('\n');break;
            case 6:write(get_val(get_rank(x+1))),putchar('\n');break;
        }
    }
    return 0;
}
