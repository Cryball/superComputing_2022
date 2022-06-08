#pragma once

#include <condition_variable>
#include <functional>
#include <queue>
#include <future>


class ThreadPool
{
public:
	explicit ThreadPool(unsigned threadsNumber = std::thread::hardware_concurrency())
		: stop(false)
	{
		if (threadsNumber == 0)
		{
			throw std::invalid_argument("More than zero threads expected");
		}

		executors.reserve(threadsNumber);

		for (unsigned i = 0; i < threadsNumber; i++)
		{
			executors.emplace_back(
				[this]
				{
					while (true)
					{
						std::function<void()> task;
						{
							std::unique_lock<std::mutex> lock(queueMutex);
							condition.wait(lock, [this] {return stop || !tasks.empty(); });

							if (stop && tasks.empty())
							{
								return;
							}

							task = std::move(tasks.front());
							tasks.pop();
						}
						task();
					}
				});
		}
	}

	template<typename F, class... Args>
	auto Enqueue(F && f, Args && ...args)
	{
		using ResultType = decltype(f(args...));

		auto task = std::make_shared<std::packaged_task<ResultType()>>(
			std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

		std::future<ResultType> result = task->get_future();
		{
			std::unique_lock lock(queueMutex);

			if (stop)
			{
				throw std::runtime_error("Enqueue on stopped ThreadPool");
			}

			tasks.emplace([task]() { (*task)(); });
		}
		condition.notify_one();
		return result;
	}

	~ThreadPool()
	{
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			stop = true;
		}

		condition.notify_all();
		for (auto & worker : executors)
		{
			worker.join();
		}
	}

private:
	std::vector<std::thread> executors;
	std::queue<std::function<void()>> tasks;

	std::mutex queueMutex;
	std::condition_variable condition;
	bool stop;
};
