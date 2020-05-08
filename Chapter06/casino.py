import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(10)
np.random.seed(10)


class Roulette:
    """Base class that simulates a roulette"""
    def __init__(self):
        self.numbers = list(range(1, 37))
        self.odds = len(self.numbers) - 1

    def spin(self):
        """
        Spins the roulette ball
        Returns:
            a random pocket
        """
        return random.choice(self.numbers)

    def bet(self, number, amount):
        """
        Bets `amount` on the `number`
        Args:
            number: number to bet on
            amount: the amount to bet

        Returns:
            reward: odds * amount of the bet is correct
            else -amount
        """
        return self.odds * amount \
            if self.spin() == number \
            else -amount

    def random_bet(self):
        """
        Randomly places a bet on one of the available choices
        Returns:
            reward: odds * amount of the bet is correct
            else -amount
        """
        choice = random.choice(self.numbers)
        return self.bet(choice, 10)

    def __str__(self):
        raise NotImplementedError


class FairRoulette(Roulette):
    """
    A fair roulette that has 1 - 36 choices
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Fair Roulette"


class AmericanRoulette(Roulette):
    """
    An unfair American roulette that has 1 - 36 choices
    and two additional choices: "0", and "00". The winning
    odds are unchanged
    """
    def __init__(self):
        super().__init__()
        # adding extra pockets, but this
        # will not change odds!
        self.numbers.append("0")
        self.numbers.append("00")

    def __str__(self):
        return "American Roulette"


class EuropeanRoulette(Roulette):
    """
    An unfair European roulette that has 1 - 36 choices
    and one additional choice: "0". The winning odds are
    unchanged
    """
    def __init__(self):
        super().__init__()
        # adding extra pocket, but this
        # will not change odds!
        self.numbers.append("0")

    def __str__(self):
        return "European Roulette"


def play_roulette(roulette: Roulette, coins=1000,
                  spins=100, trials=20):
    """
    Plays a roulette game
    Args:
        roulette: a Fair, American, or European roulette
        coins: the initial amount to start off with
        spins: the number of spins to perform in a game
        trials: the number of games to play

    Returns:

    """
    print(f"Playing {roulette}: {trials} times and {spins} spins")
    print(f"Starting off with ${coins}")
    overall_returns = []
    for trial in range(trials):
        current_game_returns = [coins]
        this_game_coins = coins
        for game in range(spins):
            ret = roulette.random_bet()
            this_game_coins += ret
            current_game_returns.append(this_game_coins)
        plt.plot(current_game_returns, alpha=0.6)
        overall_returns.append(np.mean(current_game_returns))
    print(f"Return on Investment: {(np.mean(overall_returns) - 1000)/10:.2f}%")
    plt.title(f"{roulette} with {spins} spins with {trials} trials")
    plt.xlabel("Number of spins")
    plt.ylabel("Coins")
    plt.xlim((0, spins))
    plt.show()


if __name__ == '__main__':
    play_roulette(FairRoulette(), spins=100000, trials=100)
