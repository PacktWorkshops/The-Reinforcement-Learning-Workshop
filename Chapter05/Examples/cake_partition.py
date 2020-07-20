PRICES = ["NA", 9, 40, 50, 70, 80]


def partition(cake_size):
    """
    Partitions a cake into different sizes, and calculates the
    most profitable cut configuration
    Args:
        cake_size: size of the cake

    Returns:
        the best profit possible
    """
    if cake_size == 0:
        return 0
    best_profit = -1
    for i in range(1, cake_size + 1):
        best_profit = max(best_profit, PRICES[i] + partition(cake_size - i))
    print(f"Best profit for size {cake_size} is {best_profit}")
    return best_profit


def memoized_partition(cake_size, memo):
    """
        Partitions a cake into different sizes, and calculates the
        most profitable cut configuration using memoization.
        Args:
            cake_size: size of the cake
            memo: a dictionary of `best_profit` values indexed
                by `cake_size`

        Returns:
            the best profit possible
        """
    if cake_size == 0:
        return 0
    if cake_size in memo:
        return memo[cake_size]
    else:
        best_profit = -1
        for i in range(1, cake_size + 1):
            best_profit = max(best_profit,
                              PRICES[i] + memoized_partition(cake_size - i,
                                                             memo))
        print(f"Best profit for size {cake_size} is {best_profit}")
        memo[cake_size] = best_profit
        return best_profit


def tabular_partition(cake_size):
    """
    Partitions a cake into different sizes, and calculates the
    most profitable cut configuration using tabular method.
    Args:
        cake_size: size of the cake

    Returns:
        the best profit possible

    """
    profits = [0] * (cake_size + 1)
    for i in range(1, cake_size + 1):
        best_profit = -1
        for current_size in range(1, i + 1):
            best_profit = max(best_profit,
                              PRICES[current_size] + profits[i - current_size])
        profits[i] = best_profit
    return profits[cake_size]


if __name__ == '__main__':
    size = 5
    best_profit_result = tabular_partition(size)
    print(f"Best profit: {best_profit_result}")
