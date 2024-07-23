import './style.css'
import {useState} from "react";

export default function ImageView(props){
    let {project, load_count, preprocessed} = props;
    let [scrollPosition, setScrollPosition] = useState(0)
    if (preprocessed && !project.preprocessed_available){
        return <div>Preprocessed images not available</div>
    }
    if (!preprocessed && !project.raw_available) {
        return <div>Raw images not available</div>
    }
    return (
        <div className={"imageContainer"}>
            {props.children}
            <button disabled={scrollPosition===0} onClick={()=>setScrollPosition(scrollPosition-1)}>Up</button>
            {Array(load_count).fill(0).map((_, i)=>{
                i += scrollPosition*load_count
                return <img
                    src={`http://localhost:3001/projects/${project.name}/${preprocessed?"preprocessed":"raw"}/${i}`}
                    alt={`Image ${i}`}
                    key={i}
                />
            })}
            <button onClick={()=>setScrollPosition(scrollPosition+1)}>Down</button>
        </div>
    )
}