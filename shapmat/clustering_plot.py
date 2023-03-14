import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_cluster(CRC_cluster_df,n_cluster,figsize=(10,10),title=None,savefig=False,output_path=None):
    
    fig=plt.figure(figsize=figsize)
    scatter = plt.scatter(CRC_cluster_df['PC1'],CRC_cluster_df['PC2'],
                        s=200,c=CRC_cluster_df['cluster'],cmap='Set2',alpha=1,edgecolors='black') 
        
    handles,labels = scatter.legend_elements()
    legend = plt.legend(handles,labels,fontsize=25,loc='best',markerscale=2) #,bbox_to_anchor = (1.2, 1.03)
    # for i in range(n_cluster):
        # legend.legendHandles[i]._legmarker.set_markersize(12)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title,fontsize=30)    

    plt.xlabel(f'PC1',fontsize=25)
    plt.ylabel(f'PC2',fontsize=25)

    if savefig:
        plt.savefig(output_path,bbox_inches='tight')
    plt.show()

def plot_elbow_method(CRC_cluster_df,savefig=False,output_path=None):
    wcss = []
    for i in range(1,11):
        model = KMeans(n_clusters = i, init = "k-means++",n_init="auto")
        model.fit(CRC_cluster_df[['PC1','PC2']])
        wcss.append(model.inertia_)
    plt.figure(figsize=(5,3))
    plt.plot(range(1,11), wcss)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)
    plt.xlabel('Number of clusters',fontsize=20)
    plt.ylabel('WCSS',fontsize=22)
    
    if savefig:
        plt.savefig(output_path)
    plt.show()