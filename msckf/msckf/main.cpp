#include "map.h"
#include "benchmark.h"
#include "Engine.h"
#include "visualization.h"

int main()
{
	BenchmarkNode benchmark;
	benchmark.runFromFolder();

	Engine engine;
	engine.init(&benchmark);

	std::thread viz(run);

	engine.process();

	viz.join();

	return 0;
}