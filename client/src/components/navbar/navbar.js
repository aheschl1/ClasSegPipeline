import {NavLink} from "react-router-dom";
import './style.css'

export default function Navbar(){
    return <>
        <nav>
            <ul>
                <li>
                    <NavLink to="/">Home</NavLink>
                </li>
                <li>
                    <NavLink to="/about">About</NavLink>
                </li>
            </ul>
        </nav>
    </>
}