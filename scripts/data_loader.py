{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0012d5ae",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'yfinance'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01myfinance\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01myf\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpandas\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mpd\u001b[39;00m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mdownload_data\u001b[39m(tickers, start\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m2015-01-01\u001b[39m\u001b[38;5;124m\"\u001b[39m, end\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m2024-12-31\u001b[39m\u001b[38;5;124m\"\u001b[39m):\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'yfinance'"
     ]
    }
   ],
   "source": [
    "import yfinance as yf\n",
    "import pandas as pd\n",
    "\n",
    "def download_data(tickers, start=\"2015-01-01\", end=\"2024-12-31\"):\n",
    "    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)\n",
    "    prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers})\n",
    "    return prices\n",
    "\n",
    "def compute_returns(prices):\n",
    "    returns = prices.pct_change().dropna()\n",
    "    return returns\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
