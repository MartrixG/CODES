#include<cstdio>
const int inf=0x7fffffff;
struct node{
    int max,min,sum;
    int l,r;
    int f=0;
    node* lc;
    node* rc;
};
node ROOT;
int a[100000];
int max(int a,int b){return a>b?a:b;}
int min(int a,int b){return a<b?a:b;}
void push_down(node* root)
{
    if(root->f)
    {
        root->lc->f+=root->f;
        root->rc->f+=root->f;
        root->max+=root->f;root->min+=root->f;
        root->sum+=root->f*(root->l-root->r+1);
        root->f=0;
    }
}
void build(node* root, int l, int r)
{
    if(l==r)
    {
        root->lc=NULL;
        root->rc=NULL;
        root->max=root->min=root->sum=a[l];
        root->l=root->r=l;
    }
    else
    {
        int mid=(l+r)>>1;
        root->lc=new node;
        root->rc=new node;
        build(root->lc, l, mid);
        build(root->rc, mid+1, r);
        root->max=max(root->lc->max,root->rc->max);
        root->min=min(root->lc->min,root->rc->min);
        root->sum=root->lc->sum+root->rc->sum;
        root->l=l;
        root->r=r;
    }
}
int query_max(node* root, int l,int r)
{
    if(root->r<l||root->l>r) return -1;
    push_down(root);
    if(root->l>=l&&root->r<=r)
    {
        return root->max;
    }
    int mid=(l+r)>>1;
    return max(query_max(root->lc,l,r),query_max(root->rc,l,r));
}
int query_min(node* root, int l, int r)
{
    if(root->r<l||root->l>r) return inf;
    push_down(root);
    if(root->l>=l&&root->r<=r)
    {
        return root->min;
    }
    int mid=(l+r)>>1;
    return min(query_min(root->lc,l,r),query_min(root->rc,l,r));
}
int query_sum(node* root, int l, int r)
{
    if(root->r<l||root->l>r) return 0;
    push_down(root);
    if(root->l>=l&&root->r<=r)
    {
        return root->sum;
    }
    int mid=(l+r)>>1; 
    return query_sum(root->lc,l,r)+query_sum(root->rc,l,r);
}
void modify(node* root, int l, int r, int k)
{
    if(root->r<l||root->l>r) return;
    if(root->l>=l&&root->r<=r)
    {
        root->sum+=k*(root->r-root->l+1);
        root->f+=k;
        return;
    }
    modify(root->lc,l,r,k);
    modify(root->rc,l,r,k);
}
int main()
{
    for(int i=1;i<=10;i++)
        a[i]=i;
    build(&ROOT, 1, 10);
    //printf("%d\n",query_sum(&ROOT, 2, 6));
    //printf("%d\n",query_max(&ROOT, 1, 3));
    //printf("%d\n",query_min(&ROOT, 5, 7));
    modify(&ROOT, 2, 6, 2);
    printf("%d\n",query_sum(&ROOT, 2, 6));
    printf("%d\n",query_max(&ROOT, 1, 3));
    printf("%d\n",query_min(&ROOT, 5, 7));
}