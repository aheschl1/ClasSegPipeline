import Navbar from "../components/navbar/navbar";
import ReactMarkdown from 'react-markdown'
import {useReadme} from "../hooks/github_fetch";
import remarkGfm from "remark-gfm";

export default function About(){
    let readme = useReadme()
    return <>
        <Navbar/>
        <a href={"https://github.com/aheschl1/ClasSegPipeline"}>
            GitHub Repository
        </a>
        <ReactMarkdown rehypePlugins={[remarkGfm]}>{readme}</ReactMarkdown>
        <footer style={{
            position: "fixed",
            bottom: 0,
            width: "100%",
            textAlign: "center",
            backgroundColor: "black",
            color: "white"
        }}>ClasSeg is Implemented and designed by Andrew Heschl</footer>
    </>
}