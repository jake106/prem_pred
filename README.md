<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Issues][issues-shield]][issues-url]
[![GNU][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <a href="https://github.com/jake106/prem_pred">
    <img src="images/logo.png" alt="Logo" width="100" height="100">
  </a>

<h3 align="center">Prem Predictor</h3>

  <p align="center">
    A very basic repository of models designed to predict the results of football matches in the English Premier League. Currently lacking in a number of essential features, but I'm working on it!
    <br />
    <a href="https://github.com/jake106/prem_pred"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/jake106/prem_pred">View Demo</a>
    ·
    <a href="https://github.com/jake106/prem_pred/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/jake106/prem_pred/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

There is no way to accurately predict the outcome of multiple football games over an entire league, unexpected results will always happen, and thats what makes it such a joy to watch! But what if we could at least do an ok job at prediction? Well the bookmakers can, so let's give it a try.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Prerequisites

This project is written in Python version 3.8.5. The required modules can be installed following the instructions below.

* pip
  ```sh
  pip install pandas==2.0.3 numpy==1.20.3 scipy==1.7.0 matplotlib==3.3.1
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/jake106/prem_pred.git
   ```
2. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin your_github_username/prem_pred
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Currently all functions of the package can be accessed by running the `main.py` function from the base directory with the appropriate flag. This will automatically download all training data when called for the first time. Functionality is as follows:

```bash
usage: main.py [-h] [--plot] [--simple] [--extended] [--forecast] [--simulate] [--all] [--evaluate]

Run all models.

optional arguments:
  -h, --help  show this help message and exit
  --plot      Perform some EDA - plot a few features of the dataset.
  --simple    Just train and evaluate the simple model.
  --extended  Just train and evaluate the model with seasonal extensions.
  --forecast  Forecast the results of a simulation set of data using pre-trained models. Model type
              must be specified with forecast flag
  --simulate  Simulate a league table using pre-trained models.
  --all       Run entire sequence.
  --evaluate  When actual results of prediction dataset available, run evaluation.
  --nofetch   Flag to not fetch latest match results, and instead only use older data.
```

To run the example (predicting the probability of Aston Villa finishing in the top 5 in the 24/25 Premier League), run `main.py` with the `--simulate` flag.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

### Version 0
- [ ] Add sanity check to extended model - extended model performance is lower than expected
- [ ] Add weights to training data such that previous team performance matters much less than performance in the current league
- [ ] Add season start and end dates to better capture proportion of way through a season a match is played
- [ ] Implement the ability to simulate a match result based on a simple query
- [ ] Automate evaluations scripts to only evaluate models up to latest data fetch
- [ ] Improve documentation
    - [ ] Finish README
        - [ ] Add more examples for usage as features are added
    - [ ] Improve logo
  
### Version 1.0 onward
- [ ] Add prediction models for corners
- [ ] Add flexibility to predict matches in other leagues
- [ ] Add framework to fetch bookies odds and compare with predictions
- [ ] Automate changing seasons so the model requires less reconfiguring between seasons
- [ ] Simulate betting strategies
    - [ ] Add backtesting framework

See the [open issues](https://github.com/jake106/prem_pred/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/jake106/prem_pred/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=jake106/prem_pred" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the GNU General Public License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [README Template](https://github.com/othneildrew/Best-README-Template/tree/main?tab=readme-ov-file)
* [Double-poisson football score model](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9574.1982.tb00782.x)
* [Historical football data](https://www.football-data.co.uk/englandm.php)
* [Future match data](https://fixturedownload.com/results/epl-2024)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/jake106/prem_pred.svg?style=for-the-badge
[contributors-url]: https://github.com/jake106/prem_pred/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/jake106/prem_pred.svg?style=for-the-badge
[forks-url]: https://github.com/jake106/prem_pred/network/members
[stars-shield]: https://img.shields.io/github/stars/jake106/prem_pred.svg?style=for-the-badge
[stars-url]: https://github.com/jake106/prem_pred/stargazers
[issues-shield]: https://img.shields.io/github/issues/jake106/prem_pred.svg?style=for-the-badge
[issues-url]: https://github.com/jake106/prem_pred/issues
[license-shield]: https://img.shields.io/github/license/jake106/prem_pred.svg?style=for-the-badge
[license-url]: https://github.com/jake106/prem_pred/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 


