from pathlib import Path
import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

data = {'N': 1000, 't': 1000, 'dt': 0.01, 'D': 1.0,
        'inter': 50, 'trail': 0}

def show_menu():
    click.echo("\t\t<Меню>")
    click.echo("1. Текущие параметры")
    click.echo("2. Ввод параметров")
    click.echo("3. Расчёт")
    click.echo("4. Выход")

def show_data():
    click.clear()
    click.echo("\t\t<Текущие параметры>")
    names = {
        'N': 'Количество частиц',
        't': 'Количество шагов',
        'dt': 'Шаг по времени',
        'D': 'Коэффициент диффузии',
        'inter': 'Интервал кадров, мс',
        'trail': 'Длина следа (0 — весь путь)',
    }
    for key, name in names.items():
        click.echo(f"{name}:\t{data[key]}")

def input_data():
    click.clear()
    click.echo("Ввод параметров\n\t<ENTER без ввода значения — оставить текущее>")
    data['N'] = click.prompt('Количество частиц, int', default=data['N'], type=click.IntRange(min=1))
    data['dt'] = click.prompt('Шаг по времени, float', default=data['dt'], type=click.FloatRange(min=1e-12))
    data['t'] = click.prompt('Количество шагов, int', default=data['t'], type=click.IntRange(min=1))
    data['D'] = click.prompt('Коэффициент диффузии, float', default=data['D'], type=click.FloatRange(min=1e-12))
    data['inter'] = click.prompt('Интервал кадров, мс, int', default=data['inter'], type=click.IntRange(min=1))
    data['trail'] = click.prompt('Длина следа (0 — весь путь), int', default=data['trail'], type=click.IntRange(min=0))
    click.secho("\n<Данные обновлены>")

def simulate(n, steps, dt, diffusion):
    rng = np.random.default_rng()
    increments = rng.normal(0.0, np.sqrt(2 * diffusion * dt), size=(steps, n, 2))
    coords = np.zeros((steps + 1, n, 2), dtype=np.float64)
    coords[1:] = np.cumsum(increments, axis=0)
    return coords

def coord(coords, folder):
    final = coords[-1]
    click.echo('\nКонечные координаты частиц:')
    click.echo(f'{"№":>5} {"X":>14} {"Y":>14} {"R":>14}')
    for i, (x, y) in enumerate(final, 1):
        click.echo(f'{i:5d} {x:14.6f} {y:14.6f} {np.hypot(x, y):14.6f}')

def plot_trajectories(coords, folder):
    fig, ax = plt.subplots(figsize=(9, 8))
    for i in range(coords.shape[1]):
        line, = ax.plot(coords[:, i, 0], coords[:, i, 1], lw=0.9, alpha=0.5)
        ax.scatter(*coords[-1, i], color=line.get_color(), s=15)
    ax.scatter([0], [0], color='white', marker='o', s=180, zorder=99)
    ax.scatter([0], [0], color='black', marker='+', s=140, linewidths=2.5, zorder=100, label='Начало')
    ax.set(title='Траектории частиц', xlabel='X', ylabel='Y')
    ax.axis('equal')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'trajectories.png', dpi=170)
    plt.close(fig)

def plot_histograms(coords, folder):
    final = coords[-1]
    sigma = np.sqrt(2 * data['D'] * data['dt'] * data['t'])
    xgrid = np.linspace(-4 * sigma, 4 * sigma, 500)
    pdf = np.exp(-0.5 * (xgrid / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, j, label in zip(axes, (0, 1), ('X', 'Y')):
        axis.hist(final[:, j], bins='auto', density=True, alpha=0.65, label='Моделирование')
        axis.plot(xgrid, pdf, 'k-', lw=2, label='Теория')
        axis.set(xlabel=label, ylabel='Плотность вероятности', title=f'Распределение {label}')
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(folder / 'histograms.png', dpi=170)
    plt.close(fig)

def plot_msd(coords, folder):
    times = np.arange(len(coords)) * data['dt']
    msd = np.mean(np.sum((coords - coords[0:1]) ** 2, axis=2), axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, msd, label='Моделирование')
    ax.plot(times, 4 * data['D'] * times, '--', label='Теория')
    ax.set(xlabel='Время', ylabel='Средний квадрат смещения')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / 'msd.png', dpi=170)
    plt.close(fig)
    click.echo(f'Средний квадрат смещения: {msd[-1]:.6f}; теория: {4 * data["D"] * times[-1]:.6f}')

def animate(coords, folder):
    steps, n, _ = coords.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    bound = max(1.0, float(np.max(np.abs(coords))) * 1.12)
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound),
           xlabel='X', ylabel='Y', title='Броуновское движение частиц')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.25)
    colors = plt.get_cmap('turbo')(np.linspace(0, 1, n))
    lines = [ax.plot([], [], lw=1.0, color=colors[i], alpha=0.5)[0] for i in range(n)]
    points = ax.scatter(coords[0, :, 0], coords[0, :, 1], s=25, c=colors, zorder=3)
    time_label = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top')

    def update(k):
        start = 0 if data['trail'] == 0 else max(0, k - data['trail'])
        for i, line in enumerate(lines):
            line.set_data(coords[start:k + 1, i, 0], coords[start:k + 1, i, 1])
        points.set_offsets(coords[k])
        time_label.set_text(f'Шаг: {k} / {steps - 1}\nВремя t = {k * data["dt"]:.3f}')
        return [*lines, points, time_label]

    animation = FuncAnimation(fig, update, frames=steps, interval=data['inter'],
                              blit=False, repeat=False)
    animation.save(folder / 'animation.gif', writer=PillowWriter(fps=max(1, round(1000 / data['inter']))))
    plt.close(fig)

def run_calculation():
    click.clear()
    click.secho('<Расчёт>', bold=True)
    folder = Path(__file__).resolve().parent / 'results'
    folder.mkdir(parents=True, exist_ok=True)
    coords = simulate(data['N'], data['t'], data['dt'], data['D'])
    coord(coords, folder)
    plot_trajectories(coords, folder)
    plot_histograms(coords, folder)
    plot_msd(coords, folder)
    animate(coords, folder)
    click.secho(f'Результаты сохранены')

@click.command()
def main():
    click.clear()
    
    while True:
        show_menu()
        choice = click.prompt('<Ввод>', type=click.Choice(['1', '2', '3', '4']), show_choices=False)
        
        if choice == '1':
            show_data()
            input('<ENTER для возврата в меню>')
            click.clear()
        elif choice == '2':
            input_data()
            input('<ENTER для возврата в меню>')
            click.clear()
        elif choice == '3':
            run_calculation()
            input('<ENTER для возврата в меню>')
            click.clear()
        else:
            click.secho('\nРабота программы завершена')
            break

if __name__ == '__main__':
    main()
