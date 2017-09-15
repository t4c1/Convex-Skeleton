#include "algo.h"
#include <algorithm>
#include <random>
#include <tuple>
#include <string>
#include <unordered_map>

template <typename T>
void remove(vector<T>& vec, int i) {
	vec[i] = vec.back();
	vec.pop_back();
}

double calcGlobalConvexity(const vector<vector<int>>& graph, const vector<vector<int>>& dists, int c = 1, int repeats = 100) {
	vector<vector<int>> growths(repeats);
	double glob = 0;
	#pragma omp parallel for
	for (int i = 0; i < repeats; i++) {
		growths[i] = convexGrowth(graph, dists);
		#pragma omp critical
		glob += cConvexity_Xc(growths[i], graph.size(), c);
	}
	return glob / repeats;
}

void remove_edge(vector<vector<int>>& graph, int i, int j) {
	for (int n_i = 0; n_i < graph[i].size(); n_i++) {
		if (graph[i][n_i] == j) {
			remove(graph[i], n_i);
		}
	}
	for (int n_j = 0; n_j < graph[j].size(); n_j++) {
		if (graph[j][n_j] == i) {
			remove(graph[j], n_j);
		}
	}
}

double embeddedness(const vector<vector<int>>& graph, int i, int j) {
	int emb = 0;
	for (int neighbor: graph[i]) {
		if (j != neighbor && contains(graph[j], neighbor)) {
			emb++;
		}
	}
	return emb;
}

double networkClusteringCoefficient(const vector<vector<int>>& graph) {
	int triangs = 0, triplets=0;
	for(int i=0;i<graph.size();i++) {
		for(int j:graph[i]) {
			for (int neighbor : graph[i]) {
				if (j != neighbor){
					if (contains(graph[j], neighbor)) {
						triangs++;
					}
					triplets++;
				}
			}
		}
	}
	return static_cast<double>(triangs) / static_cast<double>(triplets);
}

double rel_embeddedness(const vector<vector<int>>& graph, int i, int j) {
	return embeddedness(graph, i, j) / (double)min(graph[i].size(), graph[j].size());
}

void updateEmbeddednessWithLinkRemoval(const vector<vector<int>>& graph, vector<vector<int>>& emb, int i, int j) {
	for (int neighbor : graph[i]) {
		if (contains(graph[j], neighbor)) {
			emb[i][neighbor]--;
			emb[neighbor][i]--;
		}
	}
}

double clustering_increase(const vector<vector<int>>& graph, int i, int j) {
	int triangs_i = 0, triangs_j = 0;
	for (int neigh : graph[i]) {
		for (int neigh2 : graph[i]) {
			if (neigh != neigh2 && contains(graph[neigh], neigh2)) {
				triangs_i++;
			}
		}
	}
	for (int neigh : graph[j]) {
		for (int neigh2 : graph[j]) {
			if (neigh != neigh2 && contains(graph[neigh], neigh2)) {
				triangs_j++;
			}
		}
	}
	int emb = embeddedness(graph, i, j);
	int deg_i = graph[i].size();
	int deg_j = graph[j].size();
	double clust_i = triangs_i / (double)(deg_i * (deg_i + 1));
	double clust_j = triangs_j / (double)(deg_j * (deg_j + 1));
	double clust_i2 = (triangs_i - emb) / (double)(deg_i * (deg_i - 1));
	double clust_j2 = (triangs_j - emb) / (double)(deg_j * (deg_j - 1));
	return (clust_i2 - clust_i) + (clust_j2 - clust_j);
}
double joint_clustering_increase(const vector<vector<int>>& graph, int i, int j) {
	int triangs_i = 0, triangs_j = 0;
	for (int neigh : graph[i]) {
		for (int neigh2 : graph[i]) {
			if (neigh != neigh2 && contains(graph[neigh], neigh2)) {
				triangs_i++;
			}
		}
	}
	for (int neigh : graph[j]) {
		for (int neigh2 : graph[j]) {
			if (neigh != neigh2 && contains(graph[neigh], neigh2)) {
				triangs_j++;
			}
		}
	}
	int emb = embeddedness(graph, i, j);
	int deg_i = graph[i].size();
	int deg_j = graph[j].size();
	double clust_i = triangs_i / (double)(deg_i * (deg_i + 1));
	double clust_j = triangs_j / (double)(deg_j * (deg_j + 1));
	double clust_i2 = (triangs_i - emb) / (double)(deg_i * (deg_i - 1));
	double clust_j2 = (triangs_j - emb) / (double)(deg_j * (deg_j - 1));
	return (clust_i2 - clust_i)*deg_i + (clust_j2 - clust_j)*deg_j;
}

double convexity_increase(const vector<vector<int>>& graph, int i, int j) {
	auto d1 = distances(graph);
	int repeats = 1024*4;
	auto c1=calcGlobalConvexity(graph, d1, 1, repeats);
	cout << c1 << "\r";
	vector<vector<int>> g2 = graph;
	remove_edge(g2, i, j);
	auto d2 = distances(graph);
	auto c2 = calcGlobalConvexity(g2, d2,1,repeats);
	return c2 - c1;
}

bool altPath(const vector<vector<int>>& graph, int i, int j) {
	vector<int> todo;
	unordered_set<int> visited;
	visited.insert(i);
	for (int neighbor : graph[i]) {
		if (neighbor != j) {
			todo.push_back(neighbor);
			visited.insert(neighbor);
		}
	}
	while (!todo.empty()) {
		int current = todo.back();
		todo.pop_back();
		for (int neighbor : graph[current]) {
			if (neighbor == j) {
				return true;
			}
			if (visited.insert(neighbor).second) {
				todo.push_back(neighbor);
			}
		}
	}
	return false;
}

void reduceToSkeleton(vector<vector<int>>& graph, double (*criterion) (const vector<vector<int>>&,int,int), double treshold) {
	vector<tuple<int, int>> edges;
	for (int vert1 = 0; vert1 < graph.size(); vert1++) {
		for (int j = 0; j < graph[vert1].size();j++) {
			int vert2 = graph[vert1][j];
			if (vert1 < vert2) {
				edges.push_back(make_tuple(vert1, vert2));
			}
		}
	}
	random_shuffle(edges.begin(), edges.end());
	bool working;
	do {
		working = false;
		for (int edge_i = 0; edge_i < edges.size();edge_i++) {
			int i, j;
			tie(i, j) = edges[edge_i];
			if (criterion(graph, i, j)>treshold) {
				if (altPath(graph, i, j)) {
					working = true;
					remove_edge(graph, i, j);
				}
				remove(edges, edge_i);
				edge_i--;
			}
		}
	} while (working);
}

void reduceToSkeletonOneAtATime(vector<vector<int>>& graph, double(*criterion) (const vector<vector<int>>&, int, int), int edgesToRemove) {
	vector<tuple<int, int>> edges;
	for (int vert1 = 0; vert1 < graph.size(); vert1++) {
		for (int j = 0; j < graph[vert1].size(); j++) {
			int vert2 = graph[vert1][j];
			if (vert1 < vert2) {
				edges.push_back(make_tuple(vert1, vert2));
			}
		}
	}
	random_shuffle(edges.begin(), edges.end());
	for (int n = 0; n < edgesToRemove;) {
		double best = -INFINITY;
		int bestI = 0, bestJ=0, best_idx;
		if(edges.empty()) {
			return;
		}
		for (int edge_i = 0; edge_i < edges.size(); edge_i++) {
			int i, j;
			tie(i, j) = edges[edge_i];
			double cur_crit = criterion(graph,i,j);
			if(cur_crit>best) {
				best = cur_crit;
				bestI = i;
				bestJ = j;
				best_idx = edge_i;
			}
		}
		if (altPath(graph, bestI, bestJ)) {
			remove_edge(graph, bestI, bestJ);
			n++;
		}
		remove(edges, best_idx);
	}
}

int test1() {
	//string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\karate_club.net)";
	string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\èlanek\PhysicsData)";
	vector<string> names;
	auto g = readPajek(fn+".net", &names);
	cout << "n=" << g.size() << endl;
	auto g2=reduceToLCC(g);

	int m1 = 0;
	for(auto i:g2) {
		m1 += i.size();
	}
	m1 /= 2;
	auto d1 = distances(g2);
	double c1 = calcGlobalConvexity(g2, d1);
	cout << "original m=" << m1 << " convexity=" << c1 << endl;

	cout << "n(lcc)=" << g.size() << endl;
	string crit_names[] = { "triangs", "link_clustering", "node_clustering_increase", "joint_clustering_increase" };
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 10; j++) {
			double j2 = j;
			g = g2;
			if (i == 0) {
				reduceToSkeleton(g, [](const vector<vector<int>>& g, int a, int b) -> double {return -embeddedness(g, a, b); }, -j2);
			}
			else if (i == 1) {
				j2/=10;
				reduceToSkeleton(g, [](const vector<vector<int>>& g, int a, int b) -> double {return -rel_embeddedness(g, a, b); }, -j2);
			}
			else if (i == 2) {
				j2 /= 30;
				reduceToSkeleton(g, &clustering_increase, j2);
			}
			else if (i == 3) {
				j2 /= 5;
				reduceToSkeleton(g, &joint_clustering_increase, j2);
			}

			int m2 = 0;
			for (auto i : g) {
				m2 += i.size();
			}
			m2 /= 2;
			auto d2 = distances(g);
			double c2 = calcGlobalConvexity(g, d2);

			cout << "skeleton" << crit_names[i] << "(" << to_string(j2) << ") m=" << m2 << " convexity=" << c2 << endl;

			writePajek(fn + "_" + crit_names[i] + "_" + to_string(j2)+".net", g, names);
		}
	}
	return 0;
}

struct pairhash {
public:
	template <typename T, typename U>
	std::size_t operator()(const std::pair<T, U> &x) const
	{
		return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
	}
};


int run_remove_up_to_threshold(string fn, int repeats = 10, bool save = true) {
	vector<string> names;
	auto g = readPajek(fn + ".net", &names);
	cout << "n=" << g.size() << endl;
	auto g2 = reduceToLCC(g);

	int m1 = 0;
	for (auto i : g2) {
		m1 += i.size();
	}
	m1 /= 2;
	auto d1 = distances(g2);
	double c1 = calcGlobalConvexity(g2, d1);
	double clust1 = networkClusteringCoefficient(g);
	cout << "original m=" << m1 << " convexity=" << c1 << " network clustering=" << clust1 << endl;

	cout << "n(lcc)=" << g.size() << endl;
	unordered_map<pair<int, int>, int, pairhash> edges;
	for (int j = 1; j <= 10; j++) {
		for (int a = 0; a<g.size(); a++) {
			for (int b : g2[a]) {
				if (a<b) {
					edges[make_pair(a, b)] = 0;
				}
			}
		}
		double j2 = j;
		double avg_m2=0, avg_c2 = 0, avg_clust=0;
		for (int i = 0; i < repeats; i++) {
			
			g = g2;
			reduceToSkeleton(g, [](const vector<vector<int>>& g, int a, int b) -> double {return -embeddedness(g, a, b); }, -j2);

			int m2 = 0;
			for (auto i : g) {
				m2 += i.size();
			}
			m2 /= 2;
			auto d2 = distances(g);
			double c2 = calcGlobalConvexity(g, d2);
			double clust = networkClusteringCoefficient(g);
			avg_c2 += c2;
			avg_m2 += m2;
			avg_clust += clust;
			for(int a=0;a<g.size();a++) {
				for(int b:g[a]) {
					if(a<b) {
						edges[make_pair(a, b)]++;
					}
				}
			}
			cout << i << "/" << repeats << "\r";
			if (save) {
				writePajek(fn + "_" + "_" + to_string(j) + "_" + to_string(i) + ".net", g, names);
			}
		}
		avg_c2 /= repeats;
		avg_m2 /= repeats;
		avg_clust /= repeats;

		cout << "skeleton"  << "(" << to_string(j) << ") m=" << avg_m2 << " convexity=" << avg_c2 <<" network clustering=" << avg_clust << endl;

	}
	
	return 0;
}

int run_remove_by_one(string fn, int repeats = 10, bool save=true, double frac=0.01) {
	vector<string> names;
	auto g = readPajek(fn + ".net", &names);
	cout << "n=" << g.size() << endl;
	auto g2 = reduceToLCC(g);

	int m1 = 0;
	for (auto i : g2) {
		m1 += i.size();
	}
	m1 /= 2;
	auto d1 = distances(g2);
	double c1 = calcGlobalConvexity(g2, d1);
	double clust1 = networkClusteringCoefficient(g);
	cout << "original m=" << m1 << " convexity=" << c1 << " network clustering=" << clust1 << endl;

	cout << "n(lcc)=" << g.size() << endl;
	unordered_map<pair<int, int>, int, pairhash> edges;
	for (int j = 1; j <= m1; j+= max(int(m1*frac),1)) {
		for (int a = 0; a<g.size(); a++) {
			for (int b : g2[a]) {
				if (a<b) {
					edges[make_pair(a, b)] = 0;
				}
			}
		}
		double avg_m2 = 0, avg_c2 = 0, avg_clust = 0;
		for (int i = 0; i < repeats; i++) {

			g = g2;
			reduceToSkeletonOneAtATime(g, [](const vector<vector<int>>& g, int a, int b) -> double {return -embeddedness(g, a, b); }, j);

			int m2 = 0;
			for (auto i : g) {
				m2 += i.size();
			}
			m2 /= 2;
			auto d2 = distances(g);
			double c2 = calcGlobalConvexity(g, d2);
			double clust = networkClusteringCoefficient(g);
			avg_c2 += c2;
			avg_m2 += m2;
			avg_clust += clust;
			for (int a = 0; a<g.size(); a++) {
				for (int b : g[a]) {
					if (a<b) {
						edges[make_pair(a, b)]++;
					}
				}
			}
			cout << i << "/" << repeats << "\r";
			if (save) {
				writePajek(fn + "\\" + to_string(j) + "_" + to_string(i) + ".net", g, names);
			}
		}
		avg_c2 /= repeats;
		avg_m2 /= repeats;
		avg_clust /= repeats;

		cout << "skeleton" << "(" << to_string(j) << ") m=" << avg_m2 << " convexity=" << avg_c2 << " network clustering=" << avg_clust << endl;

		if(avg_clust==0) {
			break;
		}
	}

	return 0;
}

void main(int argc, char* argv[]) {
	//string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\karate_club.net)";
	//string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\èlanek\computersci)";
	//string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\èlanek\PhysicsData)";
	//string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\èlanek\mathematics)";
	//string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\èlanek\economics)";
	//string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\èlanek\sociology)";
	//string fn = R"(F:\Users\Tadej\Documents\fax_dn\INA\èlanek\TheoreticalPhysicsData)";
	string fn;
	int repeats = 100;
	double frac = 0.01;
	if (argc <= 1) {
		cout << "Required parameter 'input' not given. Exiting." << endl;
		exit(1);
	}
	fn = argv[1];
	if(argc>2) {
		sscanf(argv[2], "%d", &repeats);
		if (argc>3) {
			sscanf(argv[3], "%lf", &frac);
		}
	}
	
	system(("mkdir " + fn).c_str());
	run_remove_by_one(fn,repeats, true,frac);
	//cout << endl << endl << endl;
	//run_remove_up_to_threshold(fn,100);
}