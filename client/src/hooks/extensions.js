import {useEffect, useState} from "react"

export const useExtensions = ()=>{
    let [extensions, setExtensions] = useState([])
    useEffect(() => {
        fetch("http://localhost:3001/extensions")
            .then(res => res.json())
            .then(data => {
                setExtensions(data)
            })
    }, []);
    console.log(extensions)
    return extensions
}