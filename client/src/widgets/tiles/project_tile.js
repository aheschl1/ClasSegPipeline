import './tilesStyles.css';

function Tile(props) {
    // console.log(project)
    let {project, onClick} = props;
    let {name, type, config, preprocessed_available, raw_available, results_available, folds} = project;

    let total_experiments = Object.entries(results_available).map((fold)=>fold.length).reduce((a, b)=>a+b);
    let description =
        `Type: ${type} - Preprocessed: ${preprocessed_available} - Raw Available: ${raw_available} - Experiments: ${total_experiments}`;
    return (
        <div className="ProjectTile">
            <h3>{name}</h3>
            <p>{description}</p>
            <button onClick={()=>onClick()}>Inspect</button>
        </div>
    )
}

export default Tile;