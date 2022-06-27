// import Skeleton from 'react-loading-skeleton'
// import 'react-loading-skeleton/dist/skeleton.css'

class Image extends React.Component {
    render() {

        this.state = {
            margin: "auto"
        }

        return (
            <div id={this.props.id}
                 className={"img-thumbnail m-4 p-0"}
                 style={{
                     width: '500px',
                     height: '500px',
                     backgroundColor: "#D3D3D3",
                     outlineColor: "#000000",
                     outlineStyle: "solid"}}/>
        )
    }
}