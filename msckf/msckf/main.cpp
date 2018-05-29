#include "map.h"
#include "benchmark.h"
#include "Engine.h"

int main()
{
	BenchmarkNode benchmark;
	benchmark.runFromFolder();

	Engine engine;
	engine.init(&benchmark);
	engine.process();

	return 0;
}