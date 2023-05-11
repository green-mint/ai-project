import { useRoutes } from "react-router-dom";

import Home from "../pages";
import Select from "../pages/select";

export default function WebRouter() {
  let routes = useRoutes([
    { path: "/", element: <Home /> },
    { path: "/select/:img", element: <Select /> },
  ]);
  return routes;
}
