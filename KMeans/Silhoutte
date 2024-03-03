x,y = make_blobs(n_samples=500,n_features=2,centers=4,cluster_std=1,center_box=[-10.0,10.0],shuffle=True,random_state=1)
clusters = [2,3,4,5,6,7]
for cl in clusters:
  fig,(ax1,ax2) = plt.subplots(1,2)
  fig.set_size_inches(18,7)
  ax1.set_xlim([-0.1,1])
  ax1.set_ylim([0,len(x)+(cl+1)*10])
  clusterer = KMeans(n_clusters=cl,random_state=10)
  sil_labels = clusterer.fit_predict(x)
  avg_sil_score = silhouette_score(x,sil_labels)
  print(f"n_cluster:{cl}\tAverage Silhouette:{avg_sil_score}")
  sil_samples = silhouette_samples(x,sil_labels)
  y_lower = 10
  for i in range(cl):
    ith_cluster = sil_samples[sil_labels==i]
    size_ith_cluster = ith_cluster.shape[0]
    ith_cluster.sort()
    y_upper = size_ith_cluster + y_lower
    color = cm.nipy_spectral(float(i)/cl)
    ax1.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster,facecolor=color,edgecolor=color,alpha=0.7)
    ax1.text(-0.05,y_lower+0.5*size_ith_cluster,str(i))
    y_lower = y_upper + 10
  ax1.set_title("Silhouette plot for different clusters")
  ax1.set_xlabel("Silhouette coefficients")
  ax1.set_ylabel("Cluster Label")
  colors = cm.nipy_spectral(sil_labels.astype(float) / cl)
  plt.axvline(avg_sil_score,color='red',linestyle='--')
  ax1.set_yticks([])
  ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])
  ax2.scatter(x[:,0],x[:,1],marker='.',s=30,lw=0,alpha=0.7,c=colors,edgecolor='k')
  centers = clusterer.cluster_centers_
  ax2.scatter(centers[:,0],centers[:,1],marker='o',c="white",alpha=1,s=200,edgecolor="k")
  for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker="$%d$" % i,c="white",alpha=1, s=50, edgecolor="k")
  ax2.set_title("The visualization of the clustered data.")
  ax2.set_xlabel("Feature space for the 1st feature")
  ax2.set_ylabel("Feature space for the 2nd feature")
  plt.suptitle(
    "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
    % cl,
    fontsize=14,
    fontweight="bold",
  )
plt.show()
